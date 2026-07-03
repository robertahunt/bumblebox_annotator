"""
Frame-by-frame video processor for batch inference with tracking and ArUco
"""

import cv2
import gc
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.spatial import cKDTree

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.instance_tracker import Detection
from core.marker_detector import MarkerDetector
from utils.validation_metrics import distance_between_masks, mask_to_simplified_polygon, polygon_to_string


@dataclass
class BeeDetectionData:
    """Data for one bee detection in one frame"""
    video_id: str
    chamber_id: int
    frame_number: int
    bee_id: int
    aruco_code: str  # "" if not detected
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    confidence: float
    centroid_x: float
    centroid_y: float
    pred_polygon: str
    distance_to_hive_pixels: Optional[float]
    num_bees_in_chamber: Optional[int]
    avg_distance_to_other_bees_pixels: Optional[float]
    distance_to_nearest_bee_pixels: Optional[float]
    avg_distance_to_nearest_2_bees_pixels: Optional[float]
    avg_distance_to_nearest_3_bees_pixels: Optional[float]


@dataclass
class ChamberFrameData:
    """Hive pixel data for one chamber in one frame"""
    video_id: str
    chamber_id: int
    frame_number: int
    hive_pixels: Optional[int]


@dataclass
class BeeTrajectory:
    """Track a bee's position across frames for velocity calculation"""
    bee_id: int
    chamber_id: int
    aruco_code: str
    positions: List[Tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)


class BatchVideoProcessor:
    """Process video frames with detection, tracking, ArUco, and spatial analysis"""
    
    def __init__(self, video_path: Path, video_id: str, bee_model, hive_model, chamber_model,
                 tracker, confidence_threshold: float, nms_iou_threshold: float,
                 enable_aruco: bool = True, output_folder: Optional[Path] = None,
                 distance_method: str = 'contour', bee_model_type: str = 'bbox',
                 compute_spatial_metrics: bool = True, store_masks: bool = False,
                 log_callback=None, stop_callback=None, timing_log_interval: int = 1,
                 cleanup_interval: int = 25, verbose_output: bool = False,
                 max_frames: Optional[int] = None, high_quality_masks: bool = False,
                 aruco_dicts: Optional[List[str]] = None,
                 aruco_params_bank: Optional[List[Dict]] = None,
                 allowed_tag_ids: Optional[List[int]] = None):
        """
        Args:
            video_path: Path to video file
            video_id: Unique identifier for this video
            bee_model: YOLO model for bee detection
            hive_model: Optional YOLO model for hive segmentation
            chamber_model: YOLO model for chamber segmentation
            tracker: Tracking algorithm instance
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: NMS IoU threshold
            enable_aruco: Whether to detect ArUco markers on bees
            output_folder: Optional output folder for debug files
            distance_method: Method for calculating mask distances ('contour', 'bbox_filter', etc.)
            bee_model_type: Type of bee model ('bbox' or 'segmentation')
            compute_spatial_metrics: Whether to calculate hive/nearest-bee spatial metrics
            store_masks: Whether to store masks for visualization (uses significant memory)
            log_callback: Optional callable for progress/timing messages
            stop_callback: Optional callable returning True when processing should stop
            timing_log_interval: Log per-frame timings every N frames
            cleanup_interval: Clear Python/CUDA caches every N frames
            verbose_output: Whether to print detailed diagnostic/timing output
            max_frames: Optional maximum number of frames to process from the start
            high_quality_masks: Request full-resolution YOLO masks for prettier visualization
            aruco_dicts: ArUco dictionaries to use for marker detection
            aruco_params_bank: Parameter sets to try for each ArUco frame
            allowed_tag_ids: Optional allowlist of ArUco tag IDs
        """
        self.video_path = video_path
        self.video_id = video_id
        self.bee_model = bee_model
        self.hive_model = hive_model
        self.chamber_model = chamber_model
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.enable_aruco = enable_aruco
        self.distance_method = distance_method
        self.bee_model_type = bee_model_type
        self.compute_spatial_metrics = compute_spatial_metrics
        self.store_masks = store_masks
        self.log_callback = log_callback
        self.stop_callback = stop_callback
        self.timing_log_interval = max(1, int(timing_log_interval))
        self.cleanup_interval = max(1, int(cleanup_interval))
        self.verbose_output = verbose_output
        self.max_frames = max_frames
        self.high_quality_masks = high_quality_masks
        self.was_stopped = False
        self.aruco_dicts = aruco_dicts
        self.aruco_params_bank = aruco_params_bank
        self.allowed_tag_ids = allowed_tag_ids
        
        # Initialize ArUco detector for bee ID tracking
        self.marker_detector = None
        if self.enable_aruco:
            self.marker_detector = MarkerDetector(
                aruco_dicts=aruco_dicts or ['4x4_50', '4x4_100', '4x4_250', '4x4_1000'],
                aruco_params=aruco_params_bank,
                allowed_tag_ids=allowed_tag_ids,
                enable_aruco=True,
                enable_qr=False,  # Disable QR codes for performance
                min_confidence=0.2,
                debug=False,  # Disabled for performance (detect_aruco_in_bee_instances is much faster)
                debug_folder=None
            )
        
        # Data storage
        self.bee_detections: List[BeeDetectionData] = []
        self.chamber_frame_data: List[ChamberFrameData] = []
        self.bee_trajectories: Dict[int, BeeTrajectory] = {}  # bee_id -> trajectory
        self.bee_to_aruco: Dict[int, str] = {}  # bee_id -> aruco_code (retroactive)
        self.aruco_to_bee: Dict[str, int] = {}  # aruco_code -> bee_id (reverse mapping)
        self.bee_frames: Dict[int, set] = defaultdict(set)  # bee_id -> set of frame_numbers
        
        # Per-frame visualization data
        self.chambers_by_frame: Dict[int, Dict] = {}  # frame_number -> chambers_detected
        self.hive_masks_by_frame: Dict[int, Dict[int, Optional[np.ndarray]]] = {}  # frame -> chamber_id -> mask
        self.bee_masks_by_frame: Dict[int, Dict[int, Optional[np.ndarray]]] = {}  # frame -> bee_id -> mask
        
        # Chamber management
        self.chamber_mapping: Dict[int, int] = {}  # aruco_id -> chamber_id (left-to-right)
        
        # Frame counter for logging
        self.frame_count = 0
        
        # Performance timing
        self.timings = defaultdict(float)  # operation -> cumulative time
        self.timing_counts = defaultdict(int)  # operation -> count
        
        # Reset tracker at start of new video
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()

    def _log(self, message: str):
        """Log to the GUI worker when available, otherwise print."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def _log_verbose(self, message: str):
        """Log detailed diagnostics only when verbose output is enabled."""
        if self.verbose_output:
            self._log(message)

    def _should_stop(self) -> bool:
        """Return whether processing has been cancelled."""
        return bool(self.stop_callback and self.stop_callback())

    def _record_timing(self, name: str, elapsed: float, frame_timings: Dict[str, float]):
        """Record cumulative and per-frame timing for an operation."""
        self.timings[name] += elapsed
        self.timing_counts[name] += 1
        frame_timings[name] = frame_timings.get(name, 0.0) + elapsed

    def _gpu_memory_text(self) -> str:
        """Return a compact CUDA memory string for logs."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            return f", GPU {allocated:.2f} GB allocated/{reserved:.2f} GB reserved, peak {peak:.2f} GB"
        return ""

    def _sync_cuda(self):
        """Synchronize CUDA so per-step timings are attributed accurately."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _cleanup_memory(self):
        """Run periodic Python and CUDA memory cleanup."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    
    def process(self) -> bool:
        """
        Process entire video
        
        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return False

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_frames is not None and total_frames > 0:
            total_frames = min(total_frames, self.max_frames)
        frame_number = 0
        
        # For progress reporting
        last_reported_percent = -1
        
        while True:
            if self.max_frames is not None and frame_number >= self.max_frames:
                break

            if self._should_stop():
                self.was_stopped = True
                self._log(f"  Stop requested before frame {frame_number + 1}")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            self.frame_count = frame_number
            
            # Report progress every 10%
            if total_frames > 0:
                percent_complete = int((frame_number / total_frames) * 100)
                if percent_complete % 10 == 0 and percent_complete != last_reported_percent:
                    self._log(f"  Progress: {frame_number}/{total_frames} frames ({percent_complete}%)")
                    last_reported_percent = percent_complete
            
            # Process this frame
            try:
                self._process_frame(frame, frame_number)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and TORCH_AVAILABLE and torch.cuda.is_available():
                    self._log(f"  ❌ CUDA out of memory at frame {frame_number}{self._gpu_memory_text()}")
                    self._cleanup_memory()
                raise

            if self._should_stop():
                self.was_stopped = True
                self._log(f"  Stop requested after frame {frame_number}")
                break
        
        cap.release()

        if self.was_stopped:
            self._finalize_processing()
            return False
        
        # Finalize data
        self._finalize_processing()
        
        return True
    
    def _process_frame(self, frame: np.ndarray, frame_number: int):
        """Process a single frame"""
        frame_start = time.perf_counter()
        frame_timings = {}

        # 1. Run chamber detection (YOLO) and establish left-to-right ordering
        t0 = time.perf_counter()
        chambers_detected = self._detect_chambers(frame)
        # Only store for visualization if requested (saves memory)
        if self.store_masks:
            self.chambers_by_frame[frame_number] = chambers_detected
        self._record_timing('chamber_detection', time.perf_counter() - t0, frame_timings)
        
        # 2. Run bee detection
        t0 = time.perf_counter()
        # Use inference_mode to prevent autograd graph buildup in GPU memory
        if TORCH_AVAILABLE:
            with torch.inference_mode():
                bee_results = self.bee_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    retina_masks=self.high_quality_masks,
                    half=torch.cuda.is_available(),
                    verbose=False
                )
        else:
            bee_results = self.bee_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_iou_threshold,
                retina_masks=self.high_quality_masks,
                verbose=False
            )
        self._sync_cuda()
        self._record_timing('bee_detection', time.perf_counter() - t0, frame_timings)
        
        # 3. Run hive detection, if available
        t0 = time.perf_counter()
        hive_results = None
        if self.hive_model is not None:
            # Use inference_mode to prevent autograd graph buildup in GPU memory
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    hive_results = self.hive_model(
                        frame,
                        conf=self.confidence_threshold,
                        iou=self.nms_iou_threshold,
                        half=torch.cuda.is_available(),
                        verbose=False
                    )
            else:
                hive_results = self.hive_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    verbose=False
                )
        self._sync_cuda()
        self._record_timing('hive_detection', time.perf_counter() - t0, frame_timings)
        
        # 4. Convert YOLO results to Detection objects
        t0 = time.perf_counter()
        bee_detections = self._yolo_to_detections(bee_results[0])
        # Delete YOLO result objects to free GPU memory immediately
        del bee_results
        self._record_timing('yolo_conversion', time.perf_counter() - t0, frame_timings)
        
        # 5. Apply tracking to assign IDs
        t0 = time.perf_counter()
        bee_detections = self._apply_tracking(bee_detections, frame_number)
        self._record_timing('tracking', time.perf_counter() - t0, frame_timings)
        
        # 6. Extract hive masks per chamber
        t0 = time.perf_counter()
        hive_masks_by_chamber = self._extract_hive_masks(
            hive_results[0] if hive_results is not None else None,
            chambers_detected
        )
        # Delete YOLO result objects to free GPU memory immediately
        if hive_results is not None:
            del hive_results
        # Only store for visualization if requested (saves memory)
        if self.store_masks:
            self.hive_masks_by_frame[frame_number] = hive_masks_by_chamber
        self._record_timing('hive_extraction', time.perf_counter() - t0, frame_timings)
        
        # 7. Save chamber frame data (hive pixels per chamber)
        for chamber_id, hive_mask in hive_masks_by_chamber.items():
            hive_pixels = int(np.sum(hive_mask > 0)) if hive_mask is not None else None
            
            self.chamber_frame_data.append(ChamberFrameData(
                video_id=self.video_id,
                chamber_id=chamber_id,
                frame_number=frame_number,
                hive_pixels=hive_pixels
            ))
        
        # 8. Assign bees to chambers, optionally calculate spatial metrics, and write bee rows.
        self._process_bee_detections(
            bee_detections, 
            frame_number, 
            chambers_detected, 
            hive_masks_by_chamber,
            frame_timings
        )
        
        # 9. Detect ArUco codes on bees (retroactive tagging)
        if self.enable_aruco and self.marker_detector is not None:
            t0 = time.perf_counter()
            self._detect_bee_aruco_codes(frame, bee_detections)
            self._record_timing('aruco_detection', time.perf_counter() - t0, frame_timings)

        frame_elapsed = time.perf_counter() - frame_start
        if self.verbose_output and frame_number % self.timing_log_interval == 0:
            top_steps = sorted(frame_timings.items(), key=lambda item: item[1], reverse=True)
            timing_text = ", ".join(f"{name}={elapsed:.3f}s" for name, elapsed in top_steps)
            self._log(
                f"  Frame {frame_number}: total={frame_elapsed:.3f}s, "
                f"detections={len(bee_detections)}, {timing_text}{self._gpu_memory_text()}"
            )

        # Drop large per-frame references as soon as all per-frame work is complete.
        del bee_detections, hive_masks_by_chamber, chambers_detected
        
        # Report timing every 100 frames
        if self.verbose_output and frame_number % 100 == 0:
            self._print_timing_stats(frame_number)
            # Periodic garbage collection and GPU cache clearing to prevent memory buildup
            self._cleanup_memory()

        if frame_number % self.cleanup_interval == 0:
            self._cleanup_memory()
            self._log_verbose(f"  [Frame {frame_number}] Memory cleanup complete{self._gpu_memory_text()}")
        
        # More aggressive cleanup every 500 frames
        if frame_number % 500 == 0:
            self._log_verbose(f"  [Frame {frame_number}] Aggressive memory cleanup...")
            # Clear stored frames if not needed for visualization
            if not self.store_masks and frame_number > 100:
                # Keep only recent frames for trajectory calculation
                frames_to_keep = set(range(frame_number - 50, frame_number + 1))
                
                # Clear old chamber and hive mask data
                old_chamber_frames = [f for f in self.chambers_by_frame.keys() if f not in frames_to_keep]
                for f in old_chamber_frames:
                    del self.chambers_by_frame[f]
                
                old_hive_frames = [f for f in self.hive_masks_by_frame.keys() if f not in frames_to_keep]
                for f in old_hive_frames:
                    del self.hive_masks_by_frame[f]
                
                old_bee_frames = [f for f in self.bee_masks_by_frame.keys() if f not in frames_to_keep]
                for f in old_bee_frames:
                    del self.bee_masks_by_frame[f]
            
            self._cleanup_memory()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                if frame_number % 1000 == 0:  # Log every 1000 frames
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    self._log_verbose(f"    GPU: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    def _detect_chambers(self, frame: np.ndarray) -> Dict[int, Dict]:
        """
        Detect chambers using YOLO chamber model and establish left-to-right ordering
        
        Returns:
            Dict mapping chamber_id -> {'mask': np.ndarray, 'bbox': [x1,y1,x2,y2], 'centroid': (x,y)}
        """
        if self.chamber_model is None:
            # No chamber model - treat entire frame as single chamber
            return {0: {
                'mask': None,
                'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                'centroid': (frame.shape[1] / 2, frame.shape[0] / 2)
            }}
        
        # Run chamber detection
        # Use inference_mode to prevent autograd graph buildup in GPU memory
        if TORCH_AVAILABLE:
            with torch.inference_mode():
                chamber_results = self.chamber_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    half=torch.cuda.is_available(),
                    verbose=False
                )
        else:
            chamber_results = self.chamber_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_iou_threshold,
                verbose=False
            )
        self._sync_cuda()
        
        if chamber_results[0].masks is None or len(chamber_results[0].masks) == 0:
            del chamber_results
            # No chambers detected - treat entire frame as single chamber
            return {0: {
                'mask': None,
                'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                'centroid': (frame.shape[1] / 2, frame.shape[0] / 2)
            }}
        
        # Extract chamber masks and compute centroids
        chamber_data = []
        for idx in range(len(chamber_results[0].masks)):
            mask = chamber_results[0].masks.data[idx].detach().cpu().numpy()
            # Resize mask to frame size if needed
            if mask.shape[:2] != chamber_results[0].orig_shape[:2]:
                mask = cv2.resize(mask, (chamber_results[0].orig_shape[1], chamber_results[0].orig_shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            
            # Get bounding box
            bbox = chamber_results[0].boxes.xyxy[idx].detach().cpu().numpy()
            
            # Calculate centroid from mask
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                centroid_y, centroid_x = coords.mean(axis=0)
                centroid = (float(centroid_x), float(centroid_y))
            else:
                # Fall back to bbox center
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            chamber_data.append({
                'mask': mask,
                'bbox': bbox.tolist(),
                'centroid': centroid
            })
        
        # Delete YOLO result objects to free GPU memory immediately
        del chamber_results
        
        # Sort chambers left-to-right by centroid X-coordinate
        chamber_data_sorted = sorted(chamber_data, key=lambda c: c['centroid'][0])
        
        # Assign chamber IDs (0, 1, 2, ...) from left to right
        chambers = {}
        for chamber_id, data in enumerate(chamber_data_sorted):
            chambers[chamber_id] = data
        
        return chambers
    
    def _assign_bee_to_chamber(self, bee_detection: Detection, chambers_detected: Dict) -> int:
        """
        Assign a bee to a chamber based on centroid location
        
        Args:
            bee_detection: Bee detection with bbox/mask
            chambers_detected: Dict of chamber_id -> chamber data (mask, bbox, centroid)
        
        Returns:
            chamber_id (int) - defaults to 0 if no chambers or no overlap
        """
        if not chambers_detected:
            return 0
        
        # Get bee centroid
        bee_centroid = self._get_centroid(bee_detection.bbox, bee_detection.mask)
        bee_x, bee_y = int(bee_centroid[0]), int(bee_centroid[1])
        
        # Check which chamber mask contains the bee's centroid
        for chamber_id, chamber_info in chambers_detected.items():
            chamber_mask = chamber_info.get('mask')
            
            # If no mask (single chamber case), assign to that chamber
            if chamber_mask is None:
                return chamber_id
            
            # Check if bee centroid falls within chamber mask
            if 0 <= bee_y < chamber_mask.shape[0] and 0 <= bee_x < chamber_mask.shape[1]:
                if chamber_mask[bee_y, bee_x] > 0:
                    return chamber_id
        
        # Fallback: assign to chamber 0 if no overlap found
        return 0
    
    def _yolo_to_detections(self, yolo_result) -> List[Detection]:
        """Convert YOLO results to Detection objects"""
        detections = []
        
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return detections
        
        # Detach tensors before converting to numpy to break autograd graph
        boxes = yolo_result.boxes.xyxy.detach().cpu().numpy()
        confidences = yolo_result.boxes.conf.detach().cpu().numpy()
        
        # Check for masks
        has_masks = yolo_result.masks is not None
        
        # Log mask availability on first frame
        if self.verbose_output and self.frame_count == 1:
            if has_masks:
                self._log(f"✓ YOLO bee model returned {len(yolo_result.masks)} segmentation masks")
            else:
                self._log(f"⚠ YOLO bee model has NO segmentation masks (detection-only model)")
                if self.enable_aruco:
                    self._log(f"  Creating rectangular masks from bounding boxes for ArUco detection...")
                else:
                    self._log(f"  ArUco disabled; bbox detections will not allocate rectangular masks.")
        
        for idx in range(len(boxes)):
            bbox = boxes[idx]
            conf = float(confidences[idx])
            
            mask = None
            if has_masks:
                mask = yolo_result.masks.data[idx].detach().cpu().numpy()
                # Resize mask to frame size if needed
                if mask.shape[:2] != yolo_result.orig_shape[:2]:
                    # Debug on first frame
                    if self.verbose_output and self.frame_count == 1 and idx == 0:
                        self._log(f"[MASK RESIZE DEBUG]")
                        self._log(f"  Original mask shape: {mask.shape}")
                        self._log(f"  yolo_result.orig_shape: {yolo_result.orig_shape}")
                        self._log(f"  Resizing to: ({yolo_result.orig_shape[1]}, {yolo_result.orig_shape[0]})")
                    mask = cv2.resize(mask, (yolo_result.orig_shape[1], yolo_result.orig_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                    if self.verbose_output and self.frame_count == 1 and idx == 0:
                        self._log(f"  Resized mask shape: {mask.shape}")
                # Convert to binary mask (0 or 255 for marker detector compatibility)
                mask = ((mask > 0.5).astype(np.uint8)) * 255
            elif self.enable_aruco and self.bee_model_type != 'segmentation':
                # Create rectangular mask from bounding box for ArUco detection
                # This allows ArUco detection to work with detection-only models
                frame_height, frame_width = yolo_result.orig_shape[:2]
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                x1, y1, x2, y2 = bbox.astype(int)
                # Clip to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                # Fill rectangular region with 255
                mask[y1:y2, x1:x2] = 255
            
            det = Detection(
                bbox=bbox,
                mask=mask,
                confidence=conf,
                source='yolo',
                instance_id=None
            )
            detections.append(det)

        if self.verbose_output and self.bee_model_type == 'segmentation':
            missing_masks = sum(1 for det in detections if det.mask is None)
            if missing_masks:
                self._log(
                    f"  ⚠ Frame {self.frame_count}: segmentation model returned "
                    f"{missing_masks}/{len(detections)} detections without masks"
                )
        
        return detections
    
    def _apply_tracking(self, detections: List[Detection], frame_number: int) -> List[Detection]:
        """Apply tracking algorithm to assign IDs"""
        if not detections:
            return detections
        
        # Different trackers have different APIs
        if hasattr(self.tracker, 'match_detections_to_tracks'):
            # ByteTrack-style API
            tracker_result = self.tracker.match_detections_to_tracks(detections, frame_number)
            tracked_detections = []
            for tracked_det, track_id in tracker_result:
                tracked_det.instance_id = track_id
                tracked_detections.append(tracked_det)
            return tracked_detections
        else:
            # SimpleIoU/Centroid-style API
            return self.tracker.update(detections)
    
    def _extract_hive_masks(self, hive_result, chambers_detected: Dict) -> Dict[int, Optional[np.ndarray]]:
        """
        Extract hive segmentation masks per chamber
        
        Returns:
            Dict mapping chamber_id -> mask array (or None)
        """
        hive_masks_by_chamber = {}

        if hive_result is None:
            chamber_ids = chambers_detected.keys() if chambers_detected else [0]
            return {chamber_id: None for chamber_id in chamber_ids}
        
        # If no chambers detected, assign all hive to chamber 0
        if not chambers_detected:
            if hive_result.masks is not None and len(hive_result.masks) > 0:
                # Combine all hive masks
                combined_mask = None
                for idx in range(len(hive_result.masks)):
                    mask = hive_result.masks.data[idx].detach().cpu().numpy()
                    if mask.shape[:2] != hive_result.orig_shape[:2]:
                        mask = cv2.resize(mask, (hive_result.orig_shape[1], hive_result.orig_shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = np.maximum(combined_mask, mask)
                
                hive_masks_by_chamber[0] = combined_mask
            else:
                hive_masks_by_chamber[0] = None
            
            return hive_masks_by_chamber
        
        # Extract all hive masks
        hive_masks = []
        if hive_result.masks is not None and len(hive_result.masks) > 0:
            for idx in range(len(hive_result.masks)):
                mask = hive_result.masks.data[idx].detach().cpu().numpy()
                if mask.shape[:2] != hive_result.orig_shape[:2]:
                    mask = cv2.resize(mask, (hive_result.orig_shape[1], hive_result.orig_shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)
                hive_masks.append(mask)
        
        # Assign each hive mask to the chamber with maximum overlap
        if hive_masks:
            for chamber_id, chamber_info in chambers_detected.items():
                chamber_mask = chamber_info.get('mask')
                
                # Combine hive masks that overlap with this chamber
                chamber_hive_mask = None
                for hive_mask in hive_masks:
                    if chamber_mask is not None:
                        # Calculate overlap
                        overlap = np.logical_and(chamber_mask > 0, hive_mask > 0).sum()
                        if overlap > 0:
                            if chamber_hive_mask is None:
                                chamber_hive_mask = hive_mask.copy()
                            else:
                                chamber_hive_mask = np.maximum(chamber_hive_mask, hive_mask)
                    else:
                        # No chamber mask (single chamber), assign all hive
                        if chamber_hive_mask is None:
                            chamber_hive_mask = hive_mask.copy()
                        else:
                            chamber_hive_mask = np.maximum(chamber_hive_mask, hive_mask)
                
                hive_masks_by_chamber[chamber_id] = chamber_hive_mask
        else:
            # No hive masks detected
            for chamber_id in chambers_detected.keys():
                hive_masks_by_chamber[chamber_id] = None
        
        return hive_masks_by_chamber
    
    def _process_bee_detections(self, bee_detections: List[Detection], frame_number: int,
                                chambers_detected: Dict, hive_masks_by_chamber: Dict[int, Optional[np.ndarray]],
                                frame_timings: Optional[Dict[str, float]] = None):
        """Process bee detections: assign to chambers, optionally calculate spatial metrics, save data"""
        if frame_timings is None:
            frame_timings = {}

        # Group bees by chamber
        t_record = time.perf_counter()
        bees_by_chamber: Dict[int, List[Detection]] = defaultdict(list)
        
        for det in bee_detections:
            # Assign bee to chamber based on centroid location
            chamber_id = self._assign_bee_to_chamber(det, chambers_detected)
            bees_by_chamber[chamber_id].append(det)
        self._record_timing('bee_recording', time.perf_counter() - t_record, frame_timings)
        
        # Pre-compute hive coordinates and KDTree for each chamber (OPTIMIZATION)
        hive_kdtrees = {}
        if self.compute_spatial_metrics:
            t_spatial = time.perf_counter()
            for chamber_id, hive_mask in hive_masks_by_chamber.items():
                if hive_mask is not None and np.sum(hive_mask) > 0:
                    # Extract hive pixel coordinates once per chamber
                    hive_coords = np.argwhere(hive_mask > 0)  # Shape: (N, 2) as (y, x)
                    if len(hive_coords) > 0:
                        # Swap to (x, y) for consistency and build KDTree
                        hive_coords_xy = hive_coords[:, [1, 0]]  # Now (x, y)
                        hive_kdtrees[chamber_id] = cKDTree(hive_coords_xy)
                    else:
                        hive_kdtrees[chamber_id] = None
                else:
                    hive_kdtrees[chamber_id] = None
            self._record_timing('spatial_metrics', time.perf_counter() - t_spatial, frame_timings)
        
        # Process each chamber
        for chamber_id, bees in bees_by_chamber.items():
            hive_kdtree = hive_kdtrees.get(chamber_id)
            
            # Calculate centroids for all bees in this chamber (vectorize)
            bee_centroids = []
            for bee in bees:
                centroid = self._get_centroid(bee.bbox, bee.mask)
                bee_centroids.append(centroid)

            if self.compute_spatial_metrics:
                t_spatial = time.perf_counter()
                distance_matrix = self._calculate_bee_distance_matrix(bees, bee_centroids)
                self._record_timing('spatial_metrics', time.perf_counter() - t_spatial, frame_timings)
            else:
                distance_matrix = None
            
            # Process each bee
            for bee_idx, bee in enumerate(bees):
                if bee.instance_id is None:
                    continue
                
                centroid = bee_centroids[bee_idx]
                centroid_x, centroid_y = centroid
                
                if self.compute_spatial_metrics:
                    t_spatial = time.perf_counter()
                    # Calculate distance to hive (using KDTree for speed) when hive data is available.
                    distance_to_hive = (
                        self._calculate_distance_to_hive_fast(centroid, hive_kdtree)
                        if hive_kdtree is not None
                        else None
                    )
                    num_bees_in_chamber = len(bees)
                    avg_distance_to_bees = self._distance_matrix_avg_to_other_bees(
                        bee_idx, distance_matrix
                    )
                    nearest_distances = self._distance_matrix_nearest_n(
                        bee_idx, distance_matrix, n_values=[1, 2, 3]
                    )
                    self._record_timing('spatial_metrics', time.perf_counter() - t_spatial, frame_timings)
                else:
                    distance_to_hive = None
                    num_bees_in_chamber = None
                    avg_distance_to_bees = None
                    nearest_distances = {1: None, 2: None, 3: None}
                
                # Get ArUco code (if detected)
                t_record = time.perf_counter()
                aruco_code = self.bee_to_aruco.get(bee.instance_id, "")
                pred_polygon = self._mask_to_polygon_string(bee.mask)
                
                # Save bee detection data
                bbox_x, bbox_y, bbox_x2, bbox_y2 = bee.bbox
                bbox_width = bbox_x2 - bbox_x
                bbox_height = bbox_y2 - bbox_y
                
                self.bee_detections.append(BeeDetectionData(
                    video_id=self.video_id,
                    chamber_id=chamber_id,
                    frame_number=frame_number,
                    bee_id=bee.instance_id,
                    aruco_code=aruco_code,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    confidence=bee.confidence,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    pred_polygon=pred_polygon,
                    distance_to_hive_pixels=distance_to_hive,
                    num_bees_in_chamber=num_bees_in_chamber,
                    avg_distance_to_other_bees_pixels=avg_distance_to_bees,
                    distance_to_nearest_bee_pixels=nearest_distances[1],
                    avg_distance_to_nearest_2_bees_pixels=nearest_distances[2],
                    avg_distance_to_nearest_3_bees_pixels=nearest_distances[3]
                ))
                self._record_timing('bee_recording', time.perf_counter() - t_record, frame_timings)
                
                # Store bee mask for visualization (only if requested to reduce memory usage)
                if self.store_masks:
                    if frame_number not in self.bee_masks_by_frame:
                        self.bee_masks_by_frame[frame_number] = {}
                    
                    # If current detection has no mask, try to preserve previous frame's mask
                    # This handles cases where YOLO inconsistently produces masks for tracked bees
                    if bee.mask is not None:
                        self.bee_masks_by_frame[frame_number][bee.instance_id] = bee.mask
                    else:
                        # Look for mask in previous frame for same bee_id
                        previous_mask = None
                        for prev_frame in range(frame_number - 1, max(0, frame_number - 10), -1):
                            if prev_frame in self.bee_masks_by_frame:
                                prev_mask = self.bee_masks_by_frame[prev_frame].get(bee.instance_id)
                                if prev_mask is not None:
                                    previous_mask = prev_mask
                                    break
                        
                        # Only store if we found a previous mask, otherwise don't store (or store None)
                        # Storing None explicitly so we don't accidentally retrieve old values
                        if previous_mask is not None:
                            self.bee_masks_by_frame[frame_number][bee.instance_id] = previous_mask
                        else:
                            # No previous mask found - store None (this will fall back to bbox in visualization)
                            self.bee_masks_by_frame[frame_number][bee.instance_id] = None

                if self.verbose_output and self.store_masks and bee.mask is None:
                    self._log(
                        f"  ⚠ Frame {frame_number}: no segmentation mask stored for "
                        f"bee ID {bee.instance_id}; visualization will use bbox if no previous mask exists"
                    )
                
                # Update trajectory for velocity calculation
                if bee.instance_id not in self.bee_trajectories:
                    self.bee_trajectories[bee.instance_id] = BeeTrajectory(
                        bee_id=bee.instance_id,
                        chamber_id=chamber_id,
                        aruco_code=aruco_code
                    )
                
                self.bee_trajectories[bee.instance_id].positions.append(
                    (frame_number, centroid_x, centroid_y)
                )
                
                # Track which frames this bee appears in
                self.bee_frames[bee.instance_id].add(frame_number)
    
    def _detect_bee_aruco_codes(self, frame: np.ndarray, bee_detections: List[Detection]):
        """
        Detect ArUco codes on individual bees using efficient full-frame detection.
        
        Strategy:
        1. Detect all ArUco codes in the full frame once
        2. Match them to bee detections based on spatial overlap
        3. Validate assignments with strict rules:
           - If ArUco in multiple boxes in this frame: reject
           - If ArUco already assigned to another bee active in this frame: reject
           - If ArUco not yet assigned and only in one bee's box: assign
           - If ArUco was assigned before but that bee is inactive: re-identify (merge tracks)
        """
        # Diagnostic logging on first frame
        if self.verbose_output and self.frame_count == 1:
            self._log(f"\n=== Frame 1 ArUco Diagnostic ===")
            self._log(f"Total detections: {len(bee_detections)}")
            
            dets_with_id = sum(1 for det in bee_detections if det.instance_id is not None)
            dets_with_mask = sum(1 for det in bee_detections if det.mask is not None)
            dets_with_both = sum(1 for det in bee_detections if det.instance_id is not None and det.mask is not None)
            
            self._log(f"  Detections with instance_id: {dets_with_id}")
            self._log(f"  Detections with mask: {dets_with_mask}")
            self._log(f"  Detections with BOTH (valid for ArUco): {dets_with_both}")
            
            if dets_with_mask > 0:
                sample = next((det for det in bee_detections if det.mask is not None), None)
                if sample:
                    self._log(f"  Sample mask: shape={sample.mask.shape}, dtype={sample.mask.dtype}, range=[{sample.mask.min()}, {sample.mask.max()}]")
                    self._log(f"  Sample has instance_id: {sample.instance_id is not None} (id={sample.instance_id})")
            self._log("=" * 35 + "\n")
        
        # Get active bee IDs in current frame
        active_bee_ids = set(det.instance_id for det in bee_detections if det.instance_id is not None)
        
        # Convert Detection objects to annotation format for marker detector
        annotations = []
        for det in bee_detections:
            if det.instance_id is None or det.mask is None:
                continue
            
            # Convert bbox to [x, y, w, h] format
            x1, y1, x2, y2 = det.bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]
            
            # Convert mask from 0/255 to 0/1 format if needed
            mask = det.mask
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            
            annotations.append({
                'instance_id': det.instance_id,
                'mask_id': det.instance_id,
                'category': 'bee',
                'bbox': bbox,
                'mask': mask
            })
        
        if not annotations:
            return
        
        # Detect all ArUco codes in frame at once and match to bees
        detections = self.marker_detector.detect_aruco_in_bee_instances(
            image=frame,
            annotations=annotations,
            reject_multiple=True  # Reject bees with multiple ArUco codes
        )
        
        # Build reverse mapping: aruco_code -> list of instance_ids in this frame
        aruco_to_instances = {}
        for instance_id, marker_result in detections.items():
            if marker_result.marker_type == 'aruco':
                aruco_code = str(int(marker_result.marker_id))
            else:
                aruco_code = str(marker_result.marker_id)
            
            if aruco_code not in aruco_to_instances:
                aruco_to_instances[aruco_code] = []
            aruco_to_instances[aruco_code].append(instance_id)
        
        # Rule 1: Reject ArUco codes matched to multiple bees in this frame
        ambiguous_codes = set()
        for aruco_code, instance_list in aruco_to_instances.items():
            if len(instance_list) > 1:
                ambiguous_codes.add(aruco_code)
                if self.verbose_output and self.frame_count <= 10:
                    self._log(f"  ⚠ Frame {self.frame_count}: ArUco {aruco_code} in {len(instance_list)} boxes (rejected): {instance_list}")
        
        # Process each detection
        for instance_id, marker_result in detections.items():
            if marker_result.marker_type == 'aruco':
                aruco_code = str(int(marker_result.marker_id))
            else:
                aruco_code = str(marker_result.marker_id)
            
            # Rule 1: Skip if this code is ambiguous in this frame (multiple boxes)
            if aruco_code in ambiguous_codes:
                continue
            
            # Rule 2: Check if ArUco already assigned to a different bee
            if aruco_code in self.aruco_to_bee:
                assigned_bee_id = self.aruco_to_bee[aruco_code]
                
                # Rule 2a: If the assigned bee is active in this frame, reject new assignment
                if assigned_bee_id in active_bee_ids:
                    if assigned_bee_id != instance_id:
                        # Different bee has this code and is active - conflict!
                        if self.verbose_output and self.frame_count <= 10:
                            self._log(f"  ⚠ Frame {self.frame_count}: ArUco {aruco_code} already on active bee {assigned_bee_id}, rejecting for bee {instance_id}")
                        continue
                    # else: same bee, no problem
                else:
                    # Rule 2b: Assigned bee is NOT active - possible re-identification
                    # Check if these two bees ever appeared together
                    if self._bees_coexisted(assigned_bee_id, instance_id):
                        # They appeared together before - they're different bees
                        # This is a conflict - reject this assignment
                        if self.verbose_output and self.frame_count <= 10:
                            self._log(f"  ⚠ Frame {self.frame_count}: ArUco {aruco_code} conflict - bees {assigned_bee_id} and {instance_id} coexisted, rejecting")
                        continue
                    else:
                        # They never coexisted - this is re-identification
                        # Merge instance_id into assigned_bee_id
                        if self.verbose_output and self.frame_count <= 10:
                            self._log(f"  🔄 Frame {self.frame_count}: Re-identification - merging bee {instance_id} into bee {assigned_bee_id} (ArUco {aruco_code})")
                        self._merge_bee_tracks(source_id=instance_id, target_id=assigned_bee_id, aruco_code=aruco_code)
                        continue
            
            # Rule 3: New assignment - ArUco not yet assigned
            if instance_id in self.bee_to_aruco:
                # Bee already has a different ArUco code
                if self.bee_to_aruco[instance_id] != aruco_code:
                    if self.verbose_output and self.frame_count <= 10:
                        self._log(f"  ⚠ Frame {self.frame_count}: Bee {instance_id} already has ArUco {self.bee_to_aruco[instance_id]}, rejecting new code {aruco_code}")
                    continue
                # else: same code, already assigned (no action needed)
            else:
                # New assignment
                self.bee_to_aruco[instance_id] = aruco_code
                self.aruco_to_bee[aruco_code] = instance_id
                if self.verbose_output and self.frame_count <= 10:
                    self._log(f"  ✓ Frame {self.frame_count}: Assigned ArUco {aruco_code} to bee {instance_id}")
    
    def _bees_coexisted(self, bee_id_a: int, bee_id_b: int) -> bool:
        """Check if two bees ever appeared in the same frame"""
        frames_a = self.bee_frames.get(bee_id_a, set())
        frames_b = self.bee_frames.get(bee_id_b, set())
        return len(frames_a & frames_b) > 0
    
    def _merge_bee_tracks(self, source_id: int, target_id: int, aruco_code: str):
        """
        Merge source bee track into target bee track (re-identification).
        Updates all past and future detections of source_id to target_id.
        """
        # Update all detections
        for detection in self.bee_detections:
            if detection.bee_id == source_id:
                detection.bee_id = target_id
                detection.aruco_code = aruco_code
        
        # Merge trajectory data
        if source_id in self.bee_trajectories:
            source_traj = self.bee_trajectories[source_id]
            
            if target_id in self.bee_trajectories:
                # Merge positions
                self.bee_trajectories[target_id].positions.extend(source_traj.positions)
                # Sort by frame number
                self.bee_trajectories[target_id].positions.sort(key=lambda x: x[0])
            else:
                # Move entire trajectory
                self.bee_trajectories[target_id] = source_traj
                self.bee_trajectories[target_id].bee_id = target_id
            
            # Update ArUco code
            self.bee_trajectories[target_id].aruco_code = aruco_code
            
            # Remove source trajectory
            del self.bee_trajectories[source_id]
        
        # Merge frame appearances
        if source_id in self.bee_frames:
            self.bee_frames[target_id].update(self.bee_frames[source_id])
            del self.bee_frames[source_id]

        # Move stored visualization masks to the merged ID as well. Without this,
        # retroactively renamed detections can point at target_id while the mask is
        # still keyed under source_id, causing the visualizer to fall back to bbox.
        for frame_masks in self.bee_masks_by_frame.values():
            if source_id not in frame_masks:
                continue

            source_mask = frame_masks.pop(source_id)
            if target_id not in frame_masks or frame_masks[target_id] is None:
                frame_masks[target_id] = source_mask
        
        # Update bee_to_aruco mapping
        if source_id in self.bee_to_aruco:
            del self.bee_to_aruco[source_id]
        self.bee_to_aruco[target_id] = aruco_code
        
        self._log_verbose(f"    Merged bee {source_id} → {target_id} (ArUco {aruco_code})")
    
    def _get_centroid(self, bbox, mask) -> Tuple[float, float]:
        """Get centroid from bbox or mask"""
        if mask is not None:
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y, x = coords.mean(axis=0)
                return (float(x), float(y))
        
        # Fall back to bbox centroid
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _mask_to_polygon_string(self, mask: Optional[np.ndarray]) -> str:
        """Convert a bee mask to a simplified polygon string for CSV export."""
        if mask is None:
            return ""

        polygon = mask_to_simplified_polygon((mask > 0).astype(np.uint8), epsilon_percent=2.0)
        return polygon_to_string(polygon)

    def _calculate_centroid_distance_matrix(self, centroids: List[Tuple[float, float]]) -> np.ndarray:
        """Calculate all pairwise centroid distances for one chamber."""
        if not centroids:
            return np.zeros((0, 0), dtype=np.float32)

        coords = np.asarray(centroids, dtype=np.float32)
        deltas = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(deltas * deltas, axis=2, dtype=np.float32))

    def _calculate_bee_distance_matrix(self, bees: List[Detection],
                                       centroids: List[Tuple[float, float]]) -> np.ndarray:
        """Calculate all pairwise bee distances using the selected spatial metric."""
        if self.distance_method == 'centroid':
            return self._calculate_centroid_distance_matrix(centroids)

        n_bees = len(bees)
        distance_matrix = np.zeros((n_bees, n_bees), dtype=np.float32)
        if n_bees <= 1:
            return distance_matrix

        centroid_matrix = self._calculate_centroid_distance_matrix(centroids)
        for i in range(n_bees):
            current_bee = bees[i]
            for j in range(i + 1, n_bees):
                other_bee = bees[j]
                if current_bee.mask is not None and other_bee.mask is not None:
                    dist = distance_between_masks(
                        current_bee.mask,
                        other_bee.mask,
                        method=self.distance_method
                    )
                else:
                    dist = centroid_matrix[i, j]

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _distance_matrix_avg_to_other_bees(self, bee_idx: int, distance_matrix: np.ndarray) -> float:
        """Average distance from one bee to every other bee in its chamber."""
        n_bees = distance_matrix.shape[0]
        if n_bees <= 1:
            return 0.0

        distances = np.delete(distance_matrix[bee_idx], bee_idx)
        return float(np.mean(distances)) if len(distances) else 0.0

    def _distance_matrix_nearest_n(self, bee_idx: int, distance_matrix: np.ndarray,
                                   n_values: List[int]) -> Dict[int, float]:
        """Average distance from one bee to its nearest N neighbors."""
        result = {n: 0.0 for n in n_values}
        n_bees = distance_matrix.shape[0]
        if n_bees <= 1:
            return result

        distances = np.delete(distance_matrix[bee_idx], bee_idx)
        if len(distances) == 0:
            return result

        sorted_distances = np.sort(distances)
        for n in n_values:
            count = min(n, len(sorted_distances))
            result[n] = float(np.mean(sorted_distances[:count])) if count > 0 else 0.0

        return result
    
    def _calculate_distance_to_hive_fast(self, bee_centroid: Tuple[float, float], 
                                         hive_kdtree: Optional[cKDTree]) -> float:
        """Calculate Euclidean distance from bee to nearest hive pixel using KDTree (FAST)"""
        if hive_kdtree is None:
            return 0.0
        
        # Query KDTree for nearest neighbor (O(log N) instead of O(N))
        distance, _ = hive_kdtree.query(bee_centroid)
        return float(distance)
    
    def _calculate_distance_to_hive(self, bee_centroid: Tuple[float, float], 
                                    hive_mask: Optional[np.ndarray]) -> float:
        """Calculate Euclidean distance from bee to nearest hive pixel (SLOW - kept for reference)"""
        if hive_mask is None or np.sum(hive_mask) == 0:
            return 0.0
        
        # Find nearest hive pixel
        hive_coords = np.argwhere(hive_mask > 0)  # Shape: (N, 2) as (y, x)
        
        if len(hive_coords) == 0:
            return 0.0
        
        bee_x, bee_y = bee_centroid
        
        # Calculate distances to all hive pixels
        distances = np.sqrt(
            (hive_coords[:, 1] - bee_x) ** 2 + 
            (hive_coords[:, 0] - bee_y) ** 2
        )
        
        return float(np.min(distances))
    
    def _calculate_avg_distance_to_other_bees(self, bee_idx: int, 
                                              bees: List[Detection]) -> float:
        """Calculate average distance to all other bees in the same chamber
        
        Uses mask-based distance when both bees have masks (segmentation model),
        otherwise falls back to centroid distance.
        """
        if len(bees) <= 1:
            return 0.0
        
        current_bee = bees[bee_idx]
        current_has_mask = current_bee.mask is not None
        
        distances = []
        for other_idx, other_bee in enumerate(bees):
            if other_idx == bee_idx:
                continue  # Skip self
            
            # Use mask-based distance if both have masks, otherwise centroid distance
            if current_has_mask and other_bee.mask is not None:
                dist = distance_between_masks(current_bee.mask, other_bee.mask, method=self.distance_method)
            else:
                # Fallback to centroid distance
                current_centroid = self._get_centroid(current_bee.bbox, current_bee.mask)
                other_centroid = self._get_centroid(other_bee.bbox, other_bee.mask)
                dist = np.sqrt(
                    (current_centroid[0] - other_centroid[0])**2 + 
                    (current_centroid[1] - other_centroid[1])**2
                )
            
            distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
    def _calculate_nearest_n_bee_distances(self, bee_idx: int, 
                                           bees: List[Detection],
                                           n_values: List[int]) -> Dict[int, float]:
        """Calculate average distance to nearest N bees
        
        Uses mask-based distance when both bees have masks (segmentation model),
        otherwise falls back to centroid distance.
        
        Args:
            bee_idx: Index of current bee
            bees: List of all bee Detection objects in chamber
            n_values: List of N values to calculate (e.g., [1, 2, 3])
            
        Returns:
            Dict mapping N -> average distance to nearest N bees
        """
        result = {n: 0.0 for n in n_values}
        
        if len(bees) <= 1:
            return result
        
        current_bee = bees[bee_idx]
        current_has_mask = current_bee.mask is not None
        
        # Calculate distances to all other bees
        distances = []
        for other_idx, other_bee in enumerate(bees):
            if other_idx == bee_idx:
                continue  # Skip self
            
            # Use mask-based distance if both have masks, otherwise centroid distance
            if current_has_mask and other_bee.mask is not None:
                dist = distance_between_masks(current_bee.mask, other_bee.mask, method=self.distance_method)
            else:
                # Fallback to centroid distance
                current_centroid = self._get_centroid(current_bee.bbox, current_bee.mask)
                other_centroid = self._get_centroid(other_bee.bbox, other_bee.mask)
                dist = np.sqrt(
                    (current_centroid[0] - other_centroid[0])**2 + 
                    (current_centroid[1] - other_centroid[1])**2
                )
            
            distances.append(dist)
        
        if len(distances) == 0:
            return result
        
        # Sort distances to get nearest bees
        sorted_distances = np.sort(distances)
        
        # Calculate for each N value
        for n in n_values:
            if n <= len(sorted_distances):
                # Average of nearest N bees
                result[n] = float(np.mean(sorted_distances[:n]))
            else:
                # Not enough bees - use all available
                result[n] = float(np.mean(sorted_distances)) if len(sorted_distances) > 0 else 0.0
        
        return result
    
    def _finalize_processing(self):
        """Finalize processing - update ArUco codes retroactively"""
        # Update all bee detections with final ArUco codes
        for detection in self.bee_detections:
            if detection.bee_id in self.bee_to_aruco:
                detection.aruco_code = self.bee_to_aruco[detection.bee_id]
        
        # Update trajectories with final ArUco codes
        for bee_id, trajectory in self.bee_trajectories.items():
            if bee_id in self.bee_to_aruco:
                trajectory.aruco_code = self.bee_to_aruco[bee_id]
        
        # Print ArUco detection summary
        total_bees = len(self.bee_trajectories)
        tagged_bees = len(self.bee_to_aruco)
        self._log_verbose(f"\nArUco Detection Summary:")
        self._log_verbose(f"  Total unique bees tracked: {total_bees}")
        self._log_verbose(f"  Bees tagged with ArUco codes: {tagged_bees}")
        if tagged_bees > 0:
            self._log_verbose(f"  ArUco codes detected: {list(self.bee_to_aruco.values())}")
        
        # Print final timing statistics
        if self.verbose_output:
            self._log(f"\n{'='*60}")
            self._log(f"FINAL PERFORMANCE STATISTICS")
            self._log(f"{'='*60}")
            self._print_timing_stats(self.frame_count, final=True)
        
        # Free memory by clearing large data structures that are no longer needed
        # (keep only the essential data: bee_detections, chamber_frame_data, bee_trajectories)
        self._log_verbose(f"\nCleaning up memory...")
        if not self.store_masks:
            # These should already be empty, but ensure they're cleared
            self.chambers_by_frame.clear()
            self.hive_masks_by_frame.clear()
            self.bee_masks_by_frame.clear()
        
        # Force garbage collection and GPU cache clearing
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_verbose(f"✓ GPU cache cleared")
    
    def _print_timing_stats(self, frame_number: int, final: bool = False):
        """Print timing statistics for performance analysis"""
        if not self.timings:
            return
        
        total_time = sum(self.timings.values())
        
        if final:
            self._log(f"Total frames processed: {frame_number}")
            self._log(f"Total processing time: {total_time:.2f}s")
            self._log(f"Average time per frame: {total_time/frame_number:.3f}s")
            self._log(f"\nTime breakdown by operation:")
        else:
            self._log(f"\n[Frame {frame_number}] Performance stats (last {frame_number} frames):")
        
        # Sort by time (descending)
        sorted_ops = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for op_name, op_time in sorted_ops:
            count = self.timing_counts[op_name]
            avg_time = op_time / count if count > 0 else 0
            percentage = (op_time / total_time * 100) if total_time > 0 else 0
            
            if final:
                self._log(f"  {op_name:20s}: {op_time:7.2f}s ({percentage:5.1f}%) - avg {avg_time*1000:6.1f}ms/frame")
            else:
                self._log(f"  {op_name:20s}: {op_time:7.2f}s ({percentage:5.1f}%) - {avg_time*1000:6.1f}ms avg")
        
        if final:
            self._log(f"{'='*60}\n")
    
    def get_bee_detections(self) -> List[BeeDetectionData]:
        """Get all bee detection data"""
        return self.bee_detections
    
    def get_chamber_frame_data(self) -> List[ChamberFrameData]:
        """Get all chamber frame data"""
        return self.chamber_frame_data
    
    def get_bee_trajectories(self) -> Dict[int, BeeTrajectory]:
        """Get all bee trajectories for velocity calculation"""
        return self.bee_trajectories    
    def get_chambers_by_frame(self) -> Dict[int, Dict]:
        """Get chamber information per frame"""
        return self.chambers_by_frame
    
    def get_hive_masks_by_frame(self) -> Dict[int, Dict[int, Optional[np.ndarray]]]:
        """Get hive masks per frame per chamber"""
        return self.hive_masks_by_frame
    
    def get_bee_masks_by_frame(self) -> Dict[int, Dict[int, Optional[np.ndarray]]]:
        """Get bee masks per frame per bee_id"""
        return self.bee_masks_by_frame
